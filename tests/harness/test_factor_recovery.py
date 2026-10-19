"""Recovery of harness factor effects (`spec/harness.md`).

Runs are simulated on the catalog shapes with base effects drawn from the prior (`simdata`). Two harness labels,
`claude-code` and `codex`, each have eight named factors, three of them exactly 0 on each head; every run's tree
holds each named factor with probability 0.4. Every tree also holds a unique `prompt:<sha8>` whose effect is drawn
from the prior: a tool that writes a new tree per task.

Each named factor's own 80 and 95% intervals are checked against its truth, for each head apart and, within a
head, for zero and nonzero effects apart: 16 intervals per head per replicate. Three forms share the runs:
- `prior`: the five nonzero effects per head and label drawn from the factor scale per replicate, as
  `tests/belief/test_belief_recovery.py` draws its nodes. The stated rates are checked on this form, for the cost
  head. The success head misses them (it pulls nonzero effects toward 0 and over-covers zeros); its measured
  coverage is reported, and its learning is checked instead.
- `hand`: the values in `TRUE`, set by hand and large against the scale (up to 2 scales). Reported with its
  measured coverage; only its zero effects are checked.
- `control`: the `prior` truths, with the records shuffled across runs before the fit, so the named factors the fit
  reads carry no signal. A negative control: coverage of prior-drawn truths can look right with no learning at all,
  so the learning measures (slope of estimate on truth, error against the no-learning error) must separate the
  `prior` form from this one, and the cost head's nonzero coverage must fail here.

Kept apart from those rates, because the data do not identify them one by one:
- the singleton prompts: their posterior sd should stay near the factor scale (they fall back);
- `hook:Stop`, on every run: it cannot be told apart from its harness label, so the sum label plus hook is
  reported with its coverage, and the hook's own width against the scale;
- `skill:p` and `skill:q`, always together: only their sum is identified, and its coverage is reported.

The fast test runs 2 replicates of 600 runs with loose bounds. `LOOPMATH_SLOW=1` runs 40 replicates of 1,200 runs
in each form, and writes the tables and every replicate as JSON to `LOOPMATH_RECOVERY_OUT` when it is set.
`factor_probe.py` beside this file runs single seeds with a choice of extras and base scale.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "belief"))
import simdata  # noqa: E402

from loopmath import harness as H  # noqa: E402
from loopmath.belief import fit as F  # noqa: E402
from loopmath.belief.design import factor_node  # noqa: E402
from loopmath.belief.forest import default_scale  # noqa: E402
from loopmath.belief.state import load  # noqa: E402

NOW = datetime.fromisoformat("2026-10-18T12:00:00-07:00")
LABELS = ("claude-code", "codex")
NAMED = tuple(f"skill:f{i}" for i in range(1, 9))
TRUE = {  # the hand form: per head and label, the effect of each named factor (log cost; success log-odds)
    "cost": {"claude-code": (0.35, -0.30, 0.25, -0.20, 0.40, 0.0, 0.0, 0.0),
             "codex": (-0.30, 0.30, 0.20, -0.35, 0.25, 0.0, 0.0, 0.0)},
    "success": {"claude-code": (0.60, -0.50, 0.40, 0.0, 0.0, 0.0, -0.45, 0.50),
                "codex": (-0.50, 0.45, 0.0, 0.0, 0.0, 0.60, -0.40, 0.50)},
}
EXTRA = {"cost": {"hook:Stop": 0.20, "skill:p": 0.30, "skill:q": -0.10},
         "success": {"hook:Stop": 0.30, "skill:p": 0.40, "skill:q": -0.20}}
P_PRESENT = 0.4
KIND = {"cost": "cost", "success": "logit"}
FORMS = ("prior", "hand", "control")
EXTRAS = ("hook", "prompt", "pq")  # hook:Stop on every run, a unique prompt per run, skill:p with skill:q


def named_truths(seed: int, form: str) -> dict:
    """Per head and label, the named factors' effects: `TRUE`, or its nonzero ones drawn from the factor scale."""
    if form == "hand":
        return TRUE
    rng = np.random.default_rng([seed, 7])  # apart from the base effects' and the runs' streams
    return {head: {label: tuple(0.0 if v == 0.0 else float(rng.normal(0.0, default_scale("hfactor", KIND[head])))
                                for v in TRUE[head][label]) for label in LABELS} for head in ("cost", "success")}


def simulate(seed: int, n_runs: int, form: str = "prior", *, extras=EXTRAS,
             scale: float = 1.0) -> tuple[list[dict], simdata.Truth, dict]:
    """The runs of one replicate. Every form and every choice of `extras` draws the same random numbers, so they
    share the base effects, the runs and the named parts of each tree. `scale` multiplies the prior scale of the
    effects `simdata` draws (the base effects and the prompts), not the named truths. `control` shuffles the
    records across runs after the outcomes are drawn."""
    rng = np.random.default_rng(seed)
    truth = simdata.Truth(np.random.default_rng(seed + 1), scale_mult=scale)
    named = named_truths(seed, form)
    for head in ("cost", "success"):
        table = truth.effects.setdefault(head, {})
        for label in LABELS:
            for f, v in zip(NAMED, named[head][label]):
                table[factor_node(label, f)] = v
            for f, v in EXTRA[head].items():
                table[factor_node(label, f)] = v
    configs = simdata.all_configs()
    docs = []
    for i in range(n_runs):
        cfg = configs[int(rng.integers(len(configs)))]
        task = simdata.make_task(rng, i, n_tasks=150)
        fs = [f for f in NAMED if rng.random() < P_PRESENT]
        prompt = f"prompt:{int(rng.integers(2 ** 32)):08x}"
        pq = rng.random() < P_PRESENT
        fs += (["hook:Stop"] if "hook" in extras else []) + ([prompt] if "prompt" in extras else [])
        fs += ["skill:p", "skill:q"] if "pq" in extras and pq else []
        rec = {"schema": H.SCHEMA, "digest": f"sha256:{i:064x}", "parent": None, "source": "sim", "factors": fs}
        docs.append(simdata.simulate_run(truth, task, cfg, f"run_rc{seed}_{i:05d}", rng, source="user", harness=rec))
    if form == "control":
        recs = [d["run"]["ext"][H.EXT_KEY] for d in docs]
        for d, j in zip(docs, np.random.default_rng([seed, 11]).permutation(n_runs)):
            d["run"]["ext"][H.EXT_KEY] = recs[int(j)]
    return docs, truth, named


def _q(d: np.ndarray) -> np.ndarray:
    return np.percentile(d, [2.5, 10.0, 90.0, 97.5])


def replicate(seed: int, n_runs: int, home: Path, form: str = "prior", *, extras=EXTRAS,
              scale: float = 1.0) -> dict:
    """One store: hits per head and stratum, the learning measures, every named estimate, and the diagnostics
    kept apart (for the extras present)."""
    docs, truth, named = simulate(seed, n_runs, form, extras=extras, scale=scale)
    t0 = time.perf_counter()
    state = load(F.fit(home, docs=docs, no_prior=True, now=NOW))
    out: dict = {"seed": seed, "form": form, "runs": n_runs, "extras": sorted(extras), "scale": scale,
                 "fit_s": time.perf_counter() - t0, "heads": {}, "learning": {}, "estimates": {}, "apart": {}}
    for head in ("cost", "success"):
        hs = state.heads[head]
        phi = hs.phi.get("hfactor")
        cells = {s: {"n": 0, "hit80": 0, "hit95": 0} for s in ("zero", "nonzero")}
        est, num, den, err, z2 = [], 0.0, 0.0, 0.0, []
        for label in LABELS:
            for f, true in zip(NAMED, named[head][label]):
                d = hs.draws[hs.index[factor_node(label, f)]]
                lo95, lo, hi, hi95 = _q(d)
                c = cells["zero" if true == 0.0 else "nonzero"]
                c["n"] += 1
                c["hit80"] += int(lo <= true <= hi)
                c["hit95"] += int(lo95 <= true <= hi95)
                m, sd = float(np.mean(d)), float(np.std(d))
                est.append([label, f, true, m, sd])
                z2.append(((m - true) / sd) ** 2)
                if true != 0.0:
                    num, den, err = num + m * true, den + true * true, err + (m - true) ** 2
        scale0 = default_scale("hfactor", KIND[head])
        out["heads"][head] = cells
        out["learning"][head] = {"phi": phi, "slope": num / den, "mse_ratio": err / den,
                                 "sd_over_scale": float(np.mean([e[4] for e in est])) / scale0,
                                 "z_rms": float(math.sqrt(np.mean(z2)))}
        out["estimates"][head] = est
        apart: dict = {}
        if "prompt" in extras:
            prompts = [n for n in hs.nodes if n.startswith("hfactor:") and "|prompt:" in n]
            sd_ratio = [float(np.std(hs.draws[hs.index[n]])) / phi for n in prompts]
            apart.update({"prompts": len(prompts), "prompt_sd_over_phi_mean": float(np.mean(sd_ratio)),
                          "prompt_sd_over_phi_min": float(np.min(sd_ratio))})
        for label in LABELS:
            a: dict = {}
            sums = []
            if "hook" in extras:
                hook = hs.draws[hs.index[factor_node(label, "hook:Stop")]]
                a["hook_sd_over_phi"] = float(np.std(hook)) / phi
                sums.append(("label_hook", hs.draws[hs.index[f"harness:{label}"]] + hook,
                             truth.effects[head][f"harness:{label}"] + EXTRA[head]["hook:Stop"]))
            if "pq" in extras:
                p, q = (hs.draws[hs.index[factor_node(label, f)]] for f in ("skill:p", "skill:q"))
                a.update({"p_sd_over_phi": float(np.std(p)) / phi, "pq_corr": float(np.corrcoef(p, q)[0, 1])})
                sums.append(("pq_sum", p + q, EXTRA[head]["skill:p"] + EXTRA[head]["skill:q"]))
            for key, draws, true in sums:
                lo95, lo, hi, hi95 = _q(draws)
                a.update({f"{key}_mean": float(np.mean(draws)), f"{key}_true": float(true),
                          f"{key}_hit80": int(lo <= true <= hi), f"{key}_hit95": int(lo95 <= true <= hi95)})
            if a:
                apart[label] = a
        out["apart"][head] = apart
    return out


def summarize(reps: list[dict]) -> dict:
    """Hits and totals per head and stratum, the rate, and its Monte Carlo standard error over replicates."""
    table: dict = {}
    for head in ("cost", "success"):
        table[head] = {}
        for stratum in ("zero", "nonzero", "all"):
            for level in ("80", "95"):
                per = []
                hits = n = 0
                for r in reps:
                    cells = r["heads"][head]
                    cs = [cells["zero"], cells["nonzero"]] if stratum == "all" else [cells[stratum]]
                    h, k = sum(c[f"hit{level}"] for c in cs), sum(c["n"] for c in cs)
                    per.append(h / k)
                    hits, n = hits + h, n + k
                se = float(np.std(per, ddof=1) / math.sqrt(len(per))) if len(per) > 1 else None
                table[head][f"{stratum}_{level}"] = {"hits": hits, "n": n, "rate": hits / n, "mc_se": se}
    def mean_se(values: list[float]) -> dict:
        se = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None
        return {"mean": float(np.mean(values)), "mc_se": se}

    learning = {head: {k: mean_se([r["learning"][head][k] for r in reps])
                       for k in ("phi", "slope", "mse_ratio", "sd_over_scale", "z_rms")} for head in ("cost", "success")}
    apart: dict = {}
    for head in ("cost", "success"):
        first = reps[0]["apart"][head]
        apart[head] = {k: mean_se([r["apart"][head][k] for r in reps]) for k in first if not isinstance(first[k], dict)}
        if "prompt_sd_over_phi_min" in first:
            apart[head]["prompt_sd_over_phi_min"] = float(np.min([r["apart"][head]["prompt_sd_over_phi_min"]
                                                                   for r in reps]))
        for label in LABELS:
            if label in first:
                apart[head][label] = {k: float(np.mean([r["apart"][head][label][k] for r in reps]))
                                      for k in first[label] if not k.endswith("_true")}
    return {"replicates": len(reps), "table": table, "learning": learning, "apart": apart,
            "fit_s_mean": float(np.mean([r["fit_s"] for r in reps]))}


def test_factor_effects_are_recovered_fast(tmp_path):
    s = summarize([replicate(seed, 600, tmp_path / f"r{seed}") for seed in (1, 2)])
    c = summarize([replicate(seed, 600, tmp_path / f"c{seed}", "control") for seed in (1, 2)])
    for head in ("cost", "success"):
        t = s["table"][head]
        assert 0.55 <= t["all_80"]["rate"] <= 0.97, (head, t)
        assert t["zero_95"]["rate"] >= 0.75, (head, t)
        assert s["apart"][head]["prompt_sd_over_phi_mean"]["mean"] >= 0.8, s["apart"][head]
    # the cost head learns the named effects, and loses them when the records are shuffled
    assert s["learning"]["cost"]["slope"]["mean"] >= 0.8 and s["learning"]["cost"]["mse_ratio"]["mean"] <= 0.2, s
    assert abs(c["learning"]["cost"]["slope"]["mean"]) <= 0.3 and c["learning"]["cost"]["mse_ratio"]["mean"] >= 0.7, c
    assert c["table"]["cost"]["nonzero_80"]["rate"] <= 0.5, c["table"]["cost"]


def test_the_forms_share_the_runs_and_differ_only_where_they_say():
    (dp, tp, np_), (dh, th, nh), (dc, tc, nc) = (simulate(3, 20, f) for f in ("prior", "hand", "control"))
    ext = [d["run"]["ext"] for d in dp]
    assert ext == [d["run"]["ext"] for d in dh]
    assert sorted(ext, key=json.dumps) == sorted((d["run"]["ext"] for d in dc), key=json.dumps) and ext != [
        d["run"]["ext"] for d in dc]
    assert [d["attempts"] for d in dp] == [d["attempts"] for d in dc]  # shuffled after the outcomes were drawn
    assert nh == TRUE and np_ != TRUE and nc == np_
    for head in ("cost", "success"):
        for label in LABELS:
            assert [v == 0.0 for v in np_[head][label]] == [v == 0.0 for v in TRUE[head][label]]
            assert tp.effects[head][f"harness:{label}"] == th.effects[head][f"harness:{label}"]
    (dn, _, _), (dx, _, _) = simulate(3, 20, "hand", extras=()), simulate(3, 20, "hand", extras=("prompt",))
    named = [[f for f in d["run"]["ext"][H.EXT_KEY]["factors"] if f.startswith("skill:f")] for d in dh]
    assert named == [d["run"]["ext"][H.EXT_KEY]["factors"] for d in dn]
    assert all(len(d["run"]["ext"][H.EXT_KEY]["factors"]) == len(n) + 1 for d, n in zip(dx, named))


@pytest.mark.skipif(os.environ.get("LOOPMATH_SLOW") != "1", reason="40 replicates of 1,200 runs; set LOOPMATH_SLOW=1")
def test_factor_effects_are_recovered_at_the_stated_rates(tmp_path):
    out: dict = {}
    for form in FORMS:
        t0 = time.perf_counter()
        reps = [replicate(seed, 1200, tmp_path / f"{form}{seed}", form) for seed in range(100, 140)]
        out[form] = {"summary": {**summarize(reps), "wall_s": time.perf_counter() - t0}, "replicates": reps}
    path = os.environ.get("LOOPMATH_RECOVERY_OUT")
    if path:
        Path(path).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for form in FORMS:
        s = out[form]["summary"]
        for head in ("cost", "success"):
            t = s["table"][head]
            assert t["all_80"]["n"] == 640 and t["zero_80"]["n"] == 240 and t["nonzero_80"]["n"] == 400
            if form != "control":  # shuffled, each prompt node takes its run's unexplained cost, so it learns
                assert t["zero_95"]["rate"] >= 0.90, (form, head, t)
                assert s["apart"][head]["prompt_sd_over_phi_mean"]["mean"] >= 0.8, (form, s["apart"][head])
    prior, control = out["prior"]["summary"], out["control"]["summary"]
    t = prior["table"]["cost"]
    for stratum in ("zero", "nonzero", "all"):  # the stated rates: the cost head only (the module docstring)
        assert 0.70 <= t[f"{stratum}_80"]["rate"] <= 0.90, (stratum, t)
        assert 0.90 <= t[f"{stratum}_95"]["rate"] <= 0.99, (stratum, t)
    # learning, against the shuffled control (bounds a few Monte Carlo errors from the first measured run)
    pl, cl = prior["learning"], control["learning"]
    assert pl["cost"]["slope"]["mean"] >= 0.9 and pl["cost"]["mse_ratio"]["mean"] <= 0.1, pl["cost"]
    assert pl["success"]["slope"]["mean"] >= 0.15, pl["success"]
    for head in ("cost", "success"):
        assert abs(cl[head]["slope"]["mean"]) <= 0.15, (head, cl[head])
        assert cl[head]["mse_ratio"]["mean"] >= 0.85, (head, cl[head])
        assert pl[head]["mse_ratio"]["mean"] < cl[head]["mse_ratio"]["mean"], (head, pl[head], cl[head])
    assert control["table"]["cost"]["nonzero_80"]["rate"] <= 0.5, control["table"]["cost"]
